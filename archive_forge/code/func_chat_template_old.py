import asyncio
import json
from threading import Lock
from typing import List, Union
from enum import Enum
import base64
from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import tiktoken
from utils.rwkv import *
from utils.log import quick_log
import global_var
def chat_template_old(model: TextRWKV, body: ChatCompletionBody, interface: str, user: str, bot: str):
    is_raven = model.rwkv_type == RWKVType.Raven
    completion_text: str = ''
    basic_system: Union[str, None] = None
    if body.presystem:
        if body.messages[0].role == Role.System:
            basic_system = body.messages[0].content
        if basic_system is None:
            completion_text = f"\nThe following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. {bot} is very intelligent, creative and friendly. {bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. {bot} likes to tell {user} a lot about herself and her opinions. {bot} usually gives {user} kind, helpful and informative advices.\n\n" if is_raven else f'{user}{interface} hi\n\n{bot}{interface} Hi. ' + 'I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n'
        else:
            if not body.messages[0].raw:
                basic_system = basic_system.replace('\r\n', '\n').replace('\r', '\n').replace('\n\n', '\n').replace('\n', ' ').strip()
            completion_text = (f'The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. ' if is_raven else f'{user}{interface} hi\n\n{bot}{interface} Hi. ') + basic_system.replace('You are', f'{bot} is' if is_raven else 'I am').replace('you are', f'{bot} is' if is_raven else 'I am').replace("You're", f'{bot} is' if is_raven else "I'm").replace("you're", f'{bot} is' if is_raven else "I'm").replace('You', f'{bot}' if is_raven else 'I').replace('you', f'{bot}' if is_raven else 'I').replace('Your', f"{bot}'s" if is_raven else 'My').replace('your', f"{bot}'s" if is_raven else 'my').replace('你', f'{bot}' if is_raven else '我') + '\n\n'
    for message in body.messages[0 if basic_system is None else 1:]:
        append_message: str = ''
        if message.role == Role.User:
            append_message = f'{user}{interface} ' + message.content
        elif message.role == Role.Assistant:
            append_message = f'{bot}{interface} ' + message.content
        elif message.role == Role.System:
            append_message = message.content
        if not message.raw:
            append_message = append_message.replace('\r\n', '\n').replace('\r', '\n').replace('\n\n', '\n').strip()
        completion_text += append_message + '\n\n'
    completion_text += f'{bot}{interface}'
    return completion_text