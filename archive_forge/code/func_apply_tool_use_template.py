import pickle
from typing import Dict, List, Literal, Union
from tokenizers import processors
from ...pipelines.conversational import Conversation
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from ...utils.versions import require_version
def apply_tool_use_template(self, conversation: Union[List[Dict[str, str]], 'Conversation'], tools: List[Dict], **kwargs) -> Union[str, List[int]]:
    '''Create a Command-R tool-use prompt.

        Once rendered, the prompt instructs the model to generate a list of actions to perform on a set of user supplied tools
        to help carry out the user's requests.

        Conceptually, this works in the same way as `apply_chat_format`, but takes an additional `tools` parameter.

        Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys and a list of available
        tools for the model to use into a prompt string, or a list of token ids.
        This method will use the tokenizer's `default_tool_use_template` template specified at the class level.
        You can override the default template using the `tool_use_template` kwarg but the quality of your results may decrease.

        Args:
            conversation (Union[List[Dict[str, str]], "Conversation"]): A Conversation object or list of dicts
                with "role" and "content" keys, representing the chat history so far.
            tools (List[Dict]): a list of tools to render into the prompt for the model to choose from.
                See an example at the bottom of the docstring.
                The format should be:
                   * name (str): The name of the tool to be called. Valid names contain only the characters a-z,
                        A-Z, 0-9, _ and must not begin with a digit.
                   * description (str): The description of what the tool does, the model uses the description to
                        choose when and how to call the function.
                   * parameter_definitions (List[Dict]): The input parameters of the tool. Accepts a dictionary
                        where the key is the name of the parameter and the value is the parameter spec.
                        Valid parameter names contain only the characters a-z, A-Z, 0-9, _ and must not begin with a digit.
                        Parameter specs are as follows:
                       * description (str): The description of the parameter.
                       * type (str): the type of the parameter - most effective for python builtin data types, such as 'str', 'bool'
                       * required: boolean: Denotes whether the parameter is always present (required) or not. Defaults to not required.
            add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                the start of an assistant message. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            **tokenizer_kwargs: Additional kwargs to pass to the tokenizer.

        Returns:
            `str`: A rendered prompt string.
            or if tokenize=True:
            `List[int]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`.

        Examples:

        ```python
        >> tokenizer = CohereTokenizerFast.from_pretrained("CohereForAI/c4ai-command-r-v01")
        >> tools = [
            {
                "name": "internet_search",
                "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
                "parameter_definitions": {
                    "query": {
                        "description": "Query to search the internet with",
                        "type": "str",
                        "required": True
                    }
                }
            },
            {
                "name': "directly_answer",
                "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
                "parameter_definitions": {}
            }
        ]
        >> conversation = [
            {"role": "user", "content": "Whats the biggest penguin in the world?"}
        ]
        >> # render the prompt, ready for user to inspect, or for input into the model:
        >> prompt = tokenizer.apply_tool_use_template(conversation, tools=tools, tokenize=False, add_generation_prompt=True)
        >> print(prompt)
        <BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
        The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

        # System Preamble
        ## Basic Rules
        You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

        # User Preamble
        ## Task and Context
        You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

        ## Available Tools
        Here is a list of tools that you have available to you:

        \\`\\`\\`python
        def internet_search(query: str) -> List[Dict]:
            """Returns a list of relevant document snippets for a textual query retrieved from the internet

            Args:
                query (str): Query to search the internet with
            """
            pass
        \\`\\`\\`

        \\`\\`\\`python
        def directly_answer() -> List[Dict]:
            """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history
            """
            pass
        \\`\\`\\`<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:
        \\`\\`\\`json
        [
            {
                "tool_name": title of the tool in the specification,
                "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters
            }
        ]\\`\\`\\`<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
        ```
        >> inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        >> outputs = model.generate(inputs, max_new_tokens=128)
        >> print(tokenizer.decode(outputs[0]))
        Action: ```json
        [
            {
                "tool_name": "internet_search",
                "parameters": {
                    "query": "biggest penguin in the world"
                }
            }
        ]
        ```
        '''
    return self.apply_chat_template(conversation, chat_template='tool_use', tools=tools, **kwargs)