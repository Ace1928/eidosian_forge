from ..construct import CString
from ..common.utils import struct_parse, elf_assert
from .sections import Section, Symbol
def has_indexes(self):
    """ Return True if at least one version definition entry has an index
            that is stored in the vna_other field.
            This information is used for symbol versioning
        """
    if self._has_indexes is None:
        self._has_indexes = False
        for _, vernaux_iter in self.iter_versions():
            for vernaux in vernaux_iter:
                if vernaux['vna_other']:
                    self._has_indexes = True
                    break
    return self._has_indexes