import base64
import inspect
import builtins
@classmethod
def obj_from_jsondict(cls, jsondict, **additional_args):
    assert len(jsondict) == 1
    for k, v in jsondict.items():
        obj_cls = cls.cls_from_jsondict_key(k)
        return obj_cls.from_jsondict(v, **additional_args)