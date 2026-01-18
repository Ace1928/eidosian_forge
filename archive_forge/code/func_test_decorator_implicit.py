from .roundtrip import YAML
def test_decorator_implicit(self):
    from srsly.ruamel_yaml import yaml_object
    yml = YAML()

    @yaml_object(yml)
    class User2(object):

        def __init__(self, name, age):
            self.name = name
            self.age = age
    ys = '\n        - !User2\n          name: Anthon\n          age: 18\n        '
    d = yml.load(ys)
    yml.dump(d, compare=ys, unordered_lines=True)