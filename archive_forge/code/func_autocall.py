from IPython.core.magic import Bunch, Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from logging import error
@skip_doctest
@line_magic
def autocall(self, parameter_s=''):
    """Make functions callable without having to type parentheses.

        Usage:

           %autocall [mode]

        The mode can be one of: 0->Off, 1->Smart, 2->Full.  If not given, the
        value is toggled on and off (remembering the previous state).

        In more detail, these values mean:

        0 -> fully disabled

        1 -> active, but do not apply if there are no arguments on the line.

        In this mode, you get::

          In [1]: callable
          Out[1]: <built-in function callable>

          In [2]: callable 'hello'
          ------> callable('hello')
          Out[2]: False

        2 -> Active always.  Even if no arguments are present, the callable
        object is called::

          In [2]: float
          ------> float()
          Out[2]: 0.0

        Note that even with autocall off, you can still use '/' at the start of
        a line to treat the first argument on the command line as a function
        and add parentheses to it::

          In [8]: /str 43
          ------> str(43)
          Out[8]: '43'

        # all-random (note for auto-testing)
        """
    valid_modes = {0: 'Off', 1: 'Smart', 2: 'Full'}

    def errorMessage() -> str:
        error = 'Valid modes: '
        for k, v in valid_modes.items():
            error += str(k) + '->' + v + ', '
        error = error[:-2]
        return error
    if parameter_s:
        if not parameter_s in map(str, valid_modes.keys()):
            error(errorMessage())
            return
        arg = int(parameter_s)
    else:
        arg = 'toggle'
    if not arg in (*list(valid_modes.keys()), 'toggle'):
        error(errorMessage())
        return
    if arg in valid_modes.keys():
        self.shell.autocall = arg
    elif self.shell.autocall:
        self._magic_state.autocall_save = self.shell.autocall
        self.shell.autocall = 0
    else:
        try:
            self.shell.autocall = self._magic_state.autocall_save
        except AttributeError:
            self.shell.autocall = self._magic_state.autocall_save = 1
    print('Automatic calling is:', list(valid_modes.values())[self.shell.autocall])