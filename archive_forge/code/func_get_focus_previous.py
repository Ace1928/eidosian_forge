from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def get_focus_previous(self):
    """Returns the previous focusable widget using either
           :attr:`focus_previous` or the :attr:`children` similar to the
           order when the ``tab`` + ``shift`` keys are triggered together.
        """
    return self._get_focus_next('focus_previous')