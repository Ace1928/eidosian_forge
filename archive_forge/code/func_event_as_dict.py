def event_as_dict(self, event):
    names = self.events_definition[event[0]]['update_names']
    updated = {a: b for a, b in zip(names, event)}
    return updated