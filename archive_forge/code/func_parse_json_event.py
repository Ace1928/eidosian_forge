import typing
def parse_json_event(json: T_JSON_DICT) -> typing.Any:
    """ Parse a JSON dictionary into a CDP event. """
    return _event_parsers[json['method']].from_json(json['params'])