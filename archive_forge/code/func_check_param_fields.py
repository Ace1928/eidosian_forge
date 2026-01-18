from ironicclient.common.i18n import _
def check_param_fields(param_name, param_fields):
    not_existing = set(param_fields) - set(field_ids)
    if not_existing:
        raise ValueError(_('%(param)s specified with value not contained in field_ids.  Unknown value(s): %(unknown)s') % {'param': param_name, 'unknown': ', '.join(not_existing)})