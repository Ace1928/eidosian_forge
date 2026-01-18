from __future__ import annotations
import abc
import hashlib
import json
import sys
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import gradio_client.utils as client_utils
from gradio import utils
from gradio.blocks import Block, BlockContext
from gradio.component_meta import ComponentMeta
from gradio.data_classes import GradioDataModel
from gradio.events import EventListener
from gradio.layouts import Form
from gradio.processing_utils import move_files_to_cache
def get_component_instance(comp: str | dict | Component, render: bool=False, unrender: bool=False) -> Component:
    """
    Returns a component instance from a string, dict, or Component object.
    Parameters:
        comp: the component to instantiate. If a string, must be the name of a component, e.g. "dropdown". If a dict, must have a "name" key, e.g. {"name": "dropdown", "choices": ["a", "b"]}. If a Component object, will be returned as is.
        render: whether to render the component. If True, renders the component (if not already rendered). If False, does not do anything.
        unrender: whether to unrender the component. If True, unrenders the the component (if already rendered) -- this is useful when constructing an Interface or ChatInterface inside of a Blocks. If False, does not do anything.
    """
    if isinstance(comp, str):
        component_obj = component(comp, render=render)
    elif isinstance(comp, dict):
        name = comp.pop('name')
        component_cls = utils.component_or_layout_class(name)
        component_obj = component_cls(**comp, render=render)
        if isinstance(component_obj, BlockContext):
            raise ValueError(f'Invalid component: {name}')
    elif isinstance(comp, Component):
        component_obj = comp
    else:
        raise ValueError(f'Component must provided as a `str` or `dict` or `Component` but is {comp}')
    if render and (not component_obj.is_rendered):
        component_obj.render()
    elif unrender and component_obj.is_rendered:
        component_obj.unrender()
    if not isinstance(component_obj, Component):
        raise TypeError(f'Expected a Component instance, but got {component_obj.__class__}')
    return component_obj