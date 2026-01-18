import pathlib
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status as Status
from pydantic import BaseModel
from utils.rwkv import *
from utils.torch import *
import global_var
def merge_model(to_model: BaseModel, from_model: BaseModel):
    from_model_fields = [x for x in from_model.dict().keys()]
    to_model_fields = [x for x in to_model.dict().keys()]
    for field_name in from_model_fields:
        if field_name in to_model_fields:
            from_value = getattr(from_model, field_name)
            if from_value is not None:
                setattr(to_model, field_name, from_value)