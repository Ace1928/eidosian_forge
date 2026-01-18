from ..common.utils import struct_parse, dwarf_assert
def iter_attr_specs(self):
    """ Iterate over the attribute specifications for the entry. Yield
            (name, form) pairs.
        """
    for attr_spec in self['attr_spec']:
        yield (attr_spec.name, attr_spec.form)