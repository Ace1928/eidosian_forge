from mlc_chat import ChatModule
import logging
def generate_response_optimized(cm, prompt):
    if cm:
        try:
            output = cm.generate(prompt=prompt, max_length=100)
            logging.info(f'Optimized generated response: {output}')
            return output
        except Exception as e:
            logging.error(f'Error in optimized response generation: {e}')
    else:
        logging.error('Optimized MLC-LLM ChatModule not initialized')