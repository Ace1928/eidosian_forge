import re
from typing import Any, Dict, List, Optional, Tuple
from ase.db.core import Database, default_key_descriptions
from ase.db.table import Table, all_columns
def create_key_descriptions(kd: KeyDescriptions) -> KeyDescriptions:
    kd = kd.copy()
    kd.update(default_key_descriptions)
    for key, (short, long, unit) in kd.items():
        if not short:
            kd[key] = (key, key, unit)
        elif not long:
            kd[key] = (short, short, unit)
    sub = re.compile('`(.)_(.)`')
    sup = re.compile('`(.*)\\^\\{?(.*?)\\}?`')
    for key, value in kd.items():
        short, long, unit = value
        unit = sub.sub('\\1<sub>\\2</sub>', unit)
        unit = sup.sub('\\1<sup>\\2</sup>', unit)
        unit = unit.replace('\\text{', '').replace('}', '')
        kd[key] = (short, long, unit)
    return kd