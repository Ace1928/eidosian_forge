import vllm
from vllm.lora.request import LoRARequest
def test_gemma_lora(gemma_lora_files):
    llm = vllm.LLM(MODEL_PATH, max_model_len=1024, enable_lora=True, max_loras=4)
    expected_lora_output = ['more important than knowledge.\nAuthor: Albert Einstein\n', 'everyone else is already taken.\nAuthor: Oscar Wilde\n', 'so little time\nAuthor: Frank Zappa\n']
    output1 = do_sample(llm, gemma_lora_files, lora_id=1)
    for i in range(len(expected_lora_output)):
        assert output1[i].startswith(expected_lora_output[i])
    output2 = do_sample(llm, gemma_lora_files, lora_id=2)
    for i in range(len(expected_lora_output)):
        assert output2[i].startswith(expected_lora_output[i])