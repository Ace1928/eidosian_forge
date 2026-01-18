from inspect import iscoroutine, isgenerator
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from warnings import warn
import attr
def fillSlots(self, **slots: 'Flattenable') -> 'Tag':
    """
        Remember the slots provided at this position in the DOM.

        During the rendering of children of this node, slots with names in
        C{slots} will be rendered as their corresponding values.

        @return: C{self}. This enables the idiom C{return tag.fillSlots(...)} in
            renderers.
        """
    if self.slotData is None:
        self.slotData = {}
    self.slotData.update(slots)
    return self