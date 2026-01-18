from __future__ import annotations
import datetime
import json
import os
import pathlib
import traceback
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from hashlib import sha1
from importlib import import_module
from inspect import getfullargspec
from pathlib import Path
from uuid import UUID
def process_decoded(self, d):
    """
        Recursive method to support decoding dicts and lists containing
        pymatgen objects.
        """
    if isinstance(d, dict):
        if '@module' in d and '@class' in d:
            modname = d['@module']
            classname = d['@class']
            if (cls_redirect := MSONable.REDIRECT.get(modname, {}).get(classname)):
                classname = cls_redirect['@class']
                modname = cls_redirect['@module']
        elif '@module' in d and '@callable' in d:
            modname = d['@module']
            objname = d['@callable']
            classname = None
            if d.get('@bound', None) is not None:
                obj = self.process_decoded(d['@bound'])
                objname = objname.split('.')[1:]
            else:
                obj = __import__(modname, globals(), locals(), [objname], 0)
                objname = objname.split('.')
            try:
                for attr in objname:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                pass
        else:
            modname = None
            classname = None
        if classname:
            if modname and modname not in ['bson.objectid', 'numpy', 'pandas', 'torch']:
                if modname == 'datetime' and classname == 'datetime':
                    try:
                        dt = datetime.datetime.strptime(d['string'], '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        dt = datetime.datetime.strptime(d['string'], '%Y-%m-%d %H:%M:%S')
                    return dt
                if modname == 'uuid' and classname == 'UUID':
                    return UUID(d['string'])
                if modname == 'pathlib' and classname == 'Path':
                    return Path(d['string'])
                mod = __import__(modname, globals(), locals(), [classname], 0)
                if hasattr(mod, classname):
                    cls_ = getattr(mod, classname)
                    data = {k: v for k, v in d.items() if not k.startswith('@')}
                    if hasattr(cls_, 'from_dict'):
                        return cls_.from_dict(data)
                    if issubclass(cls_, Enum):
                        return cls_(d['value'])
                    if pydantic is not None and issubclass(cls_, pydantic.BaseModel):
                        d = {k: self.process_decoded(v) for k, v in data.items()}
                        return cls_(**d)
                    if dataclasses is not None and (not issubclass(cls_, MSONable)) and dataclasses.is_dataclass(cls_):
                        d = {k: self.process_decoded(v) for k, v in data.items()}
                        return cls_(**d)
            elif torch is not None and modname == 'torch' and (classname == 'Tensor'):
                if 'Complex' in d['dtype']:
                    return torch.tensor([np.array(r) + np.array(i) * 1j for r, i in zip(*d['data'])]).type(d['dtype'])
                return torch.tensor(d['data']).type(d['dtype'])
            elif np is not None and modname == 'numpy' and (classname == 'array'):
                if d['dtype'].startswith('complex'):
                    return np.array([np.array(r) + np.array(i) * 1j for r, i in zip(*d['data'])], dtype=d['dtype'])
                return np.array(d['data'], dtype=d['dtype'])
            elif modname == 'pandas':
                import pandas as pd
                if classname == 'DataFrame':
                    decoded_data = MontyDecoder().decode(d['data'])
                    return pd.DataFrame(decoded_data)
                if classname == 'Series':
                    decoded_data = MontyDecoder().decode(d['data'])
                    return pd.Series(decoded_data)
            elif bson is not None and modname == 'bson.objectid' and (classname == 'ObjectId'):
                return bson.objectid.ObjectId(d['oid'])
        return {self.process_decoded(k): self.process_decoded(v) for k, v in d.items()}
    if isinstance(d, list):
        return [self.process_decoded(x) for x in d]
    return d