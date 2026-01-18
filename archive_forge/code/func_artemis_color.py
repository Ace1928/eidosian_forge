from reportlab.lib import colors
def artemis_color(self, value):
    """Artemis color (integer) to ReportLab Color object.

        Arguments:
         - value: An int representing a functional class in the Artemis
           color scheme (see www.sanger.ac.uk for a description),
           or a string from a GenBank feature annotation for the
           color which may be dot delimited (in which case the
           first value is used).

        Takes an int representing a functional class in the Artemis color
        scheme, and returns the appropriate colors.Color object
        """
    try:
        value = int(value)
    except ValueError:
        if value.count('.'):
            value = int(value.split('.', 1)[0])
        else:
            raise
    if value in self._artemis_colorscheme:
        return self._artemis_colorscheme[value][0]
    else:
        raise ValueError('Artemis color out of range: %d' % value)