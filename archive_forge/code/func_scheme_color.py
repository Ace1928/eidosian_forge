from reportlab.lib import colors
def scheme_color(self, value):
    """Map a user-defined color integer to a ReportLab Color object.

        - value: An int representing a single color in the user-defined
          color scheme

        Takes an int representing a user-defined color and returns the
        appropriate colors.Color object.
        """
    if value in self._colorscheme:
        return self._colorscheme[value][0]
    else:
        raise ValueError('Scheme color out of range: %d' % value)