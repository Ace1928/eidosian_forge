from __future__ import absolute_import, division, print_function
from collections import namedtuple
@staticmethod
def get_object_from_list(search_list, kv_list):
    """
        Get the first matched object from a list of mso object dictionaries.
        :param search_list: Objects to search through -> List.
        :param kv_list: Key/value pairs that should match in the object. -> List[KVPair(Str, Str)]
        :return: The index and details of the object. -> Item (Named Tuple)
                 Values of provided keys of all existing objects. -> List
        """

    def kv_match(kvs, item):
        return all((item.get(kv.key) == kv.value for kv in kvs))
    match = next((Item(index, item) for index, item in enumerate(search_list) if kv_match(kv_list, item)), None)
    existing = [item.get(kv.key) for item in search_list for kv in kv_list]
    return (match, existing)