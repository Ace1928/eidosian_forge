import logging
from os_ken.services.protocols.bgp.operator.views import fields
def create_dict_view_class(detail_view_class, name):
    encode = None
    if 'encode' in dir(detail_view_class):

        def encode(self):
            dict_to_flatten = {}
            for key, obj in self.model.items():
                dict_to_flatten[key] = detail_view_class(obj).encode()
            return RdyToFlattenDict(dict_to_flatten)
    return _create_collection_view(detail_view_class, name, encode, OperatorDictView)