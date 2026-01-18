from kivy.properties import ListProperty, ObservableDict, ObjectProperty
from kivy.event import EventDispatcher
from functools import partial
@property
def observable_dict(self):
    """A dictionary instance, which when modified will trigger a `data` and
        consequently an `on_data_changed` dispatch.
        """
    return partial(ObservableDict, self.__class__.data, self)