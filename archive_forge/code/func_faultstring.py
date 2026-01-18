from wsme.utils import _
@property
def faultstring(self):
    error = _('Unknown attribute for argument %(argn)s: %(attrs)s')
    if len(self.attributes) > 1:
        error = _('Unknown attributes for argument %(argn)s: %(attrs)s')
    str_attrs = ', '.join(self.attributes)
    return error % {'argn': self.fieldname, 'attrs': str_attrs}