import re
import pytest  # NOQA
from .roundtrip import dedent
def test_issue_127():
    import srsly.ruamel_yaml

    class Ref(srsly.ruamel_yaml.YAMLObject):
        yaml_constructor = srsly.ruamel_yaml.RoundTripConstructor
        yaml_representer = srsly.ruamel_yaml.RoundTripRepresenter
        yaml_tag = u'!Ref'

        def __init__(self, logical_id):
            self.logical_id = logical_id

        @classmethod
        def from_yaml(cls, loader, node):
            return cls(loader.construct_scalar(node))

        @classmethod
        def to_yaml(cls, dumper, data):
            if isinstance(data.logical_id, srsly.ruamel_yaml.scalarstring.ScalarString):
                style = data.logical_id.style
            else:
                style = None
            return dumper.represent_scalar(cls.yaml_tag, data.logical_id, style=style)
    document = dedent('    AList:\n      - !Ref One\n      - !Ref \'Two\'\n      - !Ref\n        Two and a half\n    BList: [!Ref Three, !Ref "Four"]\n    CList:\n      - Five Six\n      - \'Seven Eight\'\n    ')
    data = srsly.ruamel_yaml.round_trip_load(document, preserve_quotes=True)
    assert srsly.ruamel_yaml.round_trip_dump(data, indent=4, block_seq_indent=2) == document.replace('\n    Two and', ' Two and')