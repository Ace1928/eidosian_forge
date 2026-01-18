def process_parameter(self, optional, has_value):
    args_required = 1
    if optional:
        args_required = 0
    self._process_item(has_value, 1, args_required)