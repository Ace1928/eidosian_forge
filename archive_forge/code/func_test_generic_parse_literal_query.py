from ..generic import GenericScalar
from ..objecttype import ObjectType
from ..schema import Schema
def test_generic_parse_literal_query():
    result = schema.execute('\n        query {\n            generic(input: {\n                int: 1,\n                float: 1.1\n                boolean: true,\n                string: "str",\n                int_list: [1, 2, 3],\n                float_list: [1.1, 2.2, 3.3],\n                boolean_list: [true, false]\n                string_list: ["str1", "str2"],\n                nested_dict: {\n                    key_a: "a",\n                    key_b: "b"\n                },\n                empty_key: undefined\n            })\n        }\n        ')
    assert not result.errors
    assert result.data == {'generic': {'int': 1, 'float': 1.1, 'boolean': True, 'string': 'str', 'int_list': [1, 2, 3], 'float_list': [1.1, 2.2, 3.3], 'boolean_list': [True, False], 'string_list': ['str1', 'str2'], 'nested_dict': {'key_a': 'a', 'key_b': 'b'}, 'empty_key': None}}