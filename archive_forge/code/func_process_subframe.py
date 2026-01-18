def process_subframe(self, subframe):
    self._process_item(subframe.has_value(), subframe.args_allowed(), subframe.args_required())