import concurrent.futures
from pathlib import Path
from typing import List, NamedTuple, Optional, Union, cast
def read_and_detect(file_path: str) -> List[dict]:
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    return cast(List[dict], chardet.detect_all(rawdata))