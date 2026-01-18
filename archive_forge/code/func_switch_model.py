import pathlib
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status as Status
from pydantic import BaseModel
from utils.rwkv import *
from utils.torch import *
import global_var
@router.post('/switch-model', tags=['Configs'])
def switch_model(body: SwitchModelBody, response: Response, request: Request):
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(Status.HTTP_403_FORBIDDEN)
    if global_var.get(global_var.Model_Status) is global_var.ModelStatus.Loading:
        response.status_code = Status.HTTP_304_NOT_MODIFIED
        return
    global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
    global_var.set(global_var.Model, None)
    torch_gc()
    if body.model == '':
        return 'success'
    devices = set([x.strip().split(' ')[0].replace('cuda:0', 'cuda') for x in body.strategy.split('->')])
    print(f'Strategy Devices: {devices}')
    try:
        state_cache.enable_state_cache()
    except HTTPException:
        pass
    os.environ['RWKV_CUDA_ON'] = '1' if body.customCuda else '0'
    global_var.set(global_var.Model_Status, global_var.ModelStatus.Loading)
    try:
        global_var.set(global_var.Model, RWKV(model=body.model, strategy=body.strategy, tokenizer=body.tokenizer))
    except Exception as e:
        print(e)
        import traceback
        print(traceback.format_exc())
        quick_log(request, body, f'Exception: {e}')
        global_var.set(global_var.Model_Status, global_var.ModelStatus.Offline)
        raise HTTPException(Status.HTTP_500_INTERNAL_SERVER_ERROR, f'failed to load: {e}')
    if body.deploy:
        global_var.set(global_var.Deploy_Mode, True)
    saved_model_config = global_var.get(global_var.Model_Config)
    init_model_config = get_rwkv_config(global_var.get(global_var.Model))
    if saved_model_config is not None:
        merge_model(init_model_config, saved_model_config)
    global_var.set(global_var.Model_Config, init_model_config)
    global_var.set(global_var.Model_Status, global_var.ModelStatus.Working)
    return 'success'