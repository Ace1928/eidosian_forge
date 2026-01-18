import os
import warnings
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ChatMessage
from langchain_core.outputs import Generation, LLMResult
def get_default_label_configs(mode: Union[str, LabelStudioMode]) -> Tuple[str, LabelStudioMode]:
    """Get default Label Studio configs for the given mode.

    Parameters:
        mode: Label Studio mode ("prompt" or "chat")

    Returns: Tuple of Label Studio config and mode
    """
    _default_label_configs = {LabelStudioMode.PROMPT.value: '\n<View>\n<Style>\n    .prompt-box {\n        background-color: white;\n        border-radius: 10px;\n        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);\n        padding: 20px;\n    }\n</Style>\n<View className="root">\n    <View className="prompt-box">\n        <Text name="prompt" value="$prompt"/>\n    </View>\n    <TextArea name="response" toName="prompt"\n              maxSubmissions="1" editable="true"\n              required="true"/>\n</View>\n<Header value="Rate the response:"/>\n<Rating name="rating" toName="prompt"/>\n</View>', LabelStudioMode.CHAT.value: '\n<View>\n<View className="root">\n     <Paragraphs name="dialogue"\n               value="$prompt"\n               layout="dialogue"\n               textKey="content"\n               nameKey="role"\n               granularity="sentence"/>\n  <Header value="Final response:"/>\n    <TextArea name="response" toName="dialogue"\n              maxSubmissions="1" editable="true"\n              required="true"/>\n</View>\n<Header value="Rate the response:"/>\n<Rating name="rating" toName="dialogue"/>\n</View>'}
    if isinstance(mode, str):
        mode = LabelStudioMode(mode)
    return (_default_label_configs[mode.value], mode)