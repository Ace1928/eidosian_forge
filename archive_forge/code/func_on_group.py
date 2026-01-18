from kivy.properties import ObjectProperty, BooleanProperty
from kivy.uix.behaviors.button import ButtonBehavior
from weakref import ref
def on_group(self, *largs):
    groups = ToggleButtonBehavior.__groups
    if self._previous_group:
        group = groups[self._previous_group]
        for item in group[:]:
            if item() is self:
                group.remove(item)
                break
    group = self._previous_group = self.group
    if group not in groups:
        groups[group] = []
    r = ref(self, ToggleButtonBehavior._clear_groups)
    groups[group].append(r)