from typing import List
from ...processing_utils import ProcessorMixin
from ...utils import is_torch_available
def encode_inputs(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
    """
        This method forwards all its arguments to [`OneFormerImageProcessor.encode_inputs`] and then tokenizes the
        task_inputs. Please refer to the docstring of this method for more information.
        """
    if task_inputs is None:
        raise ValueError('You have to specify the task_input. Found None.')
    elif images is None:
        raise ValueError('You have to specify the image. Found None.')
    if not all((task in ['semantic', 'instance', 'panoptic'] for task in task_inputs)):
        raise ValueError('task_inputs must be semantic, instance, or panoptic.')
    encoded_inputs = self.image_processor.encode_inputs(images, task_inputs, segmentation_maps, **kwargs)
    if isinstance(task_inputs, str):
        task_inputs = [task_inputs]
    if isinstance(task_inputs, List) and all((isinstance(task_input, str) for task_input in task_inputs)):
        task_token_inputs = []
        for task in task_inputs:
            task_input = f'the task is {task}'
            task_token_inputs.append(task_input)
        encoded_inputs['task_inputs'] = self._preprocess_text(task_token_inputs, max_length=self.task_seq_length)
    else:
        raise TypeError('Task Inputs should be a string or a list of strings.')
    if hasattr(encoded_inputs, 'text_inputs'):
        texts_list = encoded_inputs.text_inputs
        text_inputs = []
        for texts in texts_list:
            text_input_list = self._preprocess_text(texts, max_length=self.max_seq_length)
            text_inputs.append(text_input_list.unsqueeze(0))
        encoded_inputs['text_inputs'] = torch.cat(text_inputs, dim=0)
    return encoded_inputs