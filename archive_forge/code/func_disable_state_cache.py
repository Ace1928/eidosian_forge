from typing import Any, Dict, List, Union
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
import gc
import copy
import global_var
@router.post('/disable-state-cache', tags=['State Cache'])
def disable_state_cache():
    global trie, dtrie
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    trie = None
    dtrie = {}
    gc.collect()
    print('state cache disabled')
    return 'success'