import pytest
from pytest import approx
from unittest.mock import patch
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from llama_recipes.finetuning import main
from llama_recipes.data.sampler import LengthBasedBatchSampler
@patch('llama_recipes.finetuning.torch.cuda.is_available')
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.AutoTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.get_preprocessed_dataset')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
@pytest.mark.parametrize('cuda_is_available', [True, False])
def test_finetuning_with_validation(step_lr, optimizer, get_dataset, tokenizer, get_model, train, cuda, cuda_is_available):
    kwargs = {'run_validation': True}
    get_dataset.return_value = get_fake_dataset()
    cuda.return_value = cuda_is_available
    main(**kwargs)
    assert train.call_count == 1
    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)
    if cuda_is_available:
        assert get_model.return_value.to.call_count == 1
        assert get_model.return_value.to.call_args.args[0] == 'cuda'
    else:
        assert get_model.return_value.to.call_count == 0