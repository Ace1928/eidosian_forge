def mark_component(self, component):
    if self.comp1 is None:
        self.comp1 = component
    elif self.comp2 is None:
        self.comp2 = component
    else:
        raise ValueError('Too many component hits!')