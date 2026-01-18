from __future__ import annotations
import asyncio
import json
import os
import threading
import urllib.parse
import warnings
from typing import Any
import httpx
from packaging.version import Version
import gradio
from gradio import wasm_utils
from gradio.context import Context
from gradio.utils import get_package_version
def launched_analytics(blocks: gradio.Blocks, data: dict[str, Any]) -> None:
    if not analytics_enabled():
        return
    blocks_telemetry, inputs_telemetry, outputs_telemetry, targets_telemetry, events_telemetry = ([], [], [], [], [])
    from gradio.blocks import BlockContext
    for x in list(blocks.blocks.values()):
        blocks_telemetry.append(x.get_block_name()) if isinstance(x, BlockContext) else blocks_telemetry.append(str(x))
    for x in blocks.dependencies:
        targets_telemetry = targets_telemetry + [str(blocks.blocks[y[0]]) for y in x['targets'] if y[0] in blocks.blocks]
        events_telemetry = events_telemetry + [y[1] for y in x['targets'] if y[0] in blocks.blocks]
        inputs_telemetry = inputs_telemetry + [str(blocks.blocks[y]) for y in x['inputs'] if y in blocks.blocks]
        outputs_telemetry = outputs_telemetry + [str(blocks.blocks[y]) for y in x['outputs'] if y in blocks.blocks]
    additional_data = {'version': get_package_version(), 'is_kaggle': blocks.is_kaggle, 'is_sagemaker': blocks.is_sagemaker, 'using_auth': blocks.auth is not None, 'dev_mode': blocks.dev_mode, 'show_api': blocks.show_api, 'show_error': blocks.show_error, 'title': blocks.title, 'inputs': blocks.input_components if blocks.mode == 'interface' else inputs_telemetry, 'outputs': blocks.output_components if blocks.mode == 'interface' else outputs_telemetry, 'targets': targets_telemetry, 'blocks': blocks_telemetry, 'events': events_telemetry, 'is_wasm': wasm_utils.IS_WASM}
    data.update(additional_data)
    _do_analytics_request(url=f'{ANALYTICS_URL}gradio-launched-telemetry/', data=data)