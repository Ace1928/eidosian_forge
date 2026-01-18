import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
def register_xxx(**kw):
    import srsly.ruamel_yaml as yaml

    class XXX(yaml.comments.CommentedMap):

        @staticmethod
        def yaml_dump(dumper, data):
            return dumper.represent_mapping(u'!xxx', data)

        @classmethod
        def yaml_load(cls, constructor, node):
            data = cls()
            yield data
            constructor.construct_mapping(node, data)
    yaml.add_constructor(u'!xxx', XXX.yaml_load, constructor=yaml.RoundTripConstructor)
    yaml.add_representer(XXX, XXX.yaml_dump, representer=yaml.RoundTripRepresenter)