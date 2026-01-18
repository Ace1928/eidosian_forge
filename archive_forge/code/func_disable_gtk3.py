import sys
import select
def disable_gtk3(self):
    """Disable event loop integration with PyGTK.

        This merely sets PyOS_InputHook to NULL.
        """
    self.clear_inputhook()