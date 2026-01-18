from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def image_qa(image, question):
    if 'image' not in image:
        raise ValueError(f'`image` ({image}) is not an image.')
    return f'This is the answer to {question} from {image}.'