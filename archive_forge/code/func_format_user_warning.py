import warnings
def format_user_warning(self, warning_id, message_args):
    try:
        template = self._user_warning_templates[warning_id]
    except KeyError:
        fail = 'brz warning: {!r}, {!r}'.format(warning_id, message_args)
        warnings.warn('no template for warning: ' + fail)
        return str(fail)
    try:
        return str(template) % message_args
    except ValueError as e:
        fail = 'brz unprintable warning: {!r}, {!r}, {}'.format(warning_id, message_args, e)
        warnings.warn(fail)
        return str(fail)