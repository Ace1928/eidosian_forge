import os.path
from ._paml import Paml
from . import _parse_codeml
def read_ctl_file(self, ctl_file):
    """Parse a control file and load the options into the Codeml instance."""
    temp_options = {}
    if not os.path.isfile(ctl_file):
        raise FileNotFoundError(f'File not found: {ctl_file!r}')
    else:
        with open(ctl_file) as ctl_handle:
            for line in ctl_handle:
                line = line.strip()
                uncommented = line.split('*', 1)[0]
                if uncommented != '':
                    if '=' not in uncommented:
                        raise AttributeError(f'Malformed line in control file:\n{line!r}')
                    option, value = uncommented.split('=', 1)
                    option = option.strip()
                    value = value.strip()
                    if option == 'seqfile':
                        self.alignment = value
                    elif option == 'treefile':
                        self.tree = value
                    elif option == 'outfile':
                        self.out_file = value
                    elif option == 'NSsites':
                        site_classes = value.split(' ')
                        for n in range(len(site_classes)):
                            try:
                                site_classes[n] = int(site_classes[n])
                            except ValueError:
                                raise TypeError(f'Invalid site class: {site_classes[n]}') from None
                        temp_options['NSsites'] = site_classes
                    elif option not in self._options:
                        raise KeyError(f'Invalid option: {option}')
                    else:
                        if '.' in value:
                            try:
                                converted_value = float(value)
                            except ValueError:
                                converted_value = value
                        else:
                            try:
                                converted_value = int(value)
                            except ValueError:
                                converted_value = value
                        temp_options[option] = converted_value
    for option in self._options:
        if option in temp_options:
            self._options[option] = temp_options[option]
        else:
            self._options[option] = None