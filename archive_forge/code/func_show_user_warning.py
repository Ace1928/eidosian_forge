import warnings
def show_user_warning(self, warning_id, **message_args):
    """Show a warning to the user.

        This is specifically for things that are under the user's control (eg
        outdated formats), not for internal program warnings like deprecated
        APIs.

        This can be overridden by UIFactory subclasses to show it in some
        appropriate way; the default UIFactory is noninteractive and does
        nothing.  format_user_warning maps it to a string, though other
        presentations can be used for particular UIs.

        Args:
          warning_id: An identifier like 'cross_format_fetch' used to
            check if the message is suppressed and to look up the string.
          message_args: Arguments to be interpolated into the message.
        """
    pass