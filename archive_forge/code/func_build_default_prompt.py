from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum
def build_default_prompt(agent_type: AgentType, conversations: List[ConversationTurn], llama_guard_version: LlamaGuardVersion=LlamaGuardVersion.LLAMA_GUARD_2):
    if llama_guard_version == LlamaGuardVersion.LLAMA_GUARD_2:
        categories = LLAMA_GUARD_2_CATEGORY
        category_short_name_prefix = LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX
        prompt_template = PROMPT_TEMPLATE_2
    else:
        categories = LLAMA_GUARD_1_CATEGORY
        category_short_name_prefix = LLAMA_GUARD_1_CATEGORY_SHORT_NAME_PREFIX
        prompt_template = PROMPT_TEMPLATE_1
    return build_custom_prompt(agent_type, conversations, categories, category_short_name_prefix, prompt_template)