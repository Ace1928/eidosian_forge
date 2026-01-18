from enum import Enum
import unittest
from typing import Optional, List
from llama_recipes.data.llama_guard.finetuning_data_formatter import (
def format_examples_and_compare_to_expected(self, training_examples: List[TrainingExample], expected_formatted_examples: List[str], agent_type_to_check: AgentType, formatter_configs: Optional[FormatterConfigs]=None) -> None:
    formatter_configs = formatter_configs if formatter_configs is not None else FinetuningDataFormatterTests.create_most_conservative_formatter_configs(agent_type_to_check)
    formatted_examples = create_formatted_finetuning_examples(training_examples, formatter_configs)
    assert len(formatted_examples) == len(expected_formatted_examples)
    for i in range(len(formatted_examples)):
        if formatted_examples[i] != expected_formatted_examples[i]:
            print(f'Failed on actual output {i}:')
            print(formatted_examples[i])
        assert formatted_examples[i] == expected_formatted_examples[i]