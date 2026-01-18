import warnings
def recommend_upgrade(self, current_format_name, basedir):
    """Recommend the user upgrade a control directory.

        Args:
          current_format_name: Description of the current format
          basedir: Location of the control dir
        """
    self.show_user_warning('recommend_upgrade', current_format_name=current_format_name, basedir=basedir)