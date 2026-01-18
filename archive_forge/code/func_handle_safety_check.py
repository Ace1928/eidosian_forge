import fire
import os
import sys
import time
import torch
from transformers import AutoTokenizer
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model
def handle_safety_check(are_safe_user_prompt, user_prompt, safety_results_user_prompt, are_safe_system_prompt, system_prompt, safety_results_system_prompt):
    """
    Handles the output based on the safety check of both user and system prompts.

    Parameters:
    - are_safe_user_prompt (bool): Indicates whether the user prompt is safe.
    - user_prompt (str): The user prompt that was checked for safety.
    - safety_results_user_prompt (list of tuples): A list of tuples for the user prompt containing the method, safety status, and safety report.
    - are_safe_system_prompt (bool): Indicates whether the system prompt is safe.
    - system_prompt (str): The system prompt that was checked for safety.
    - safety_results_system_prompt (list of tuples): A list of tuples for the system prompt containing the method, safety status, and safety report.
    """

    def print_safety_results(are_safe_prompt, prompt, safety_results, prompt_type='User'):
        """
        Prints the safety results for a prompt.

        Parameters:
        - are_safe_prompt (bool): Indicates whether the prompt is safe.
        - prompt (str): The prompt that was checked for safety.
        - safety_results (list of tuples): A list of tuples containing the method, safety status, and safety report.
        - prompt_type (str): The type of prompt (User/System).
        """
        if are_safe_prompt:
            print(f'{prompt_type} prompt deemed safe.')
            print(f'{prompt_type} prompt:\n{prompt}')
        else:
            print(f'{prompt_type} prompt deemed unsafe.')
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
            print(f'Skipping the inference as the {prompt_type.lower()} prompt is not safe.')
            sys.exit(1)
    print_safety_results(are_safe_user_prompt, user_prompt, safety_results_user_prompt, 'User')
    print_safety_results(are_safe_system_prompt, system_prompt, safety_results_system_prompt, 'System')