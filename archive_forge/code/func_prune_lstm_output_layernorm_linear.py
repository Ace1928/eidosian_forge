from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def prune_lstm_output_layernorm_linear(lstm: nn.LSTM, getitem: Callable, layernorm: Optional[nn.LayerNorm], linear: nn.Linear) -> None:
    for i in range(lstm.num_layers):
        if parametrize.is_parametrized(lstm, f'weight_ih_l{i}'):
            parametrization_dict = cast(nn.ModuleDict, lstm.parametrizations)
            weight_parameterizations = cast(ParametrizationList, parametrization_dict[f'weight_ih_l{i}'])
            mask = weight_parameterizations[0].mask
            with torch.no_grad():
                parametrize.remove_parametrizations(lstm, f'weight_ih_l{i}', leave_parametrized=True)
                setattr(lstm, f'weight_ih_l{i}', nn.Parameter(getattr(lstm, f'weight_ih_l{i}')[mask]))
                setattr(lstm, f'bias_ih_l{i}', nn.Parameter(getattr(lstm, f'bias_ih_l{i}')[mask]))
        if parametrize.is_parametrized(lstm, f'weight_hh_l{i}'):
            parametrization_dict = cast(nn.ModuleDict, lstm.parametrizations)
            weight_parameterizations = cast(ParametrizationList, parametrization_dict[f'weight_hh_l{i}'])
            mask = weight_parameterizations[0].mask
            with torch.no_grad():
                parametrize.remove_parametrizations(lstm, f'weight_hh_l{i}', leave_parametrized=True)
                W_hi, W_hf, W_hg, W_ho = torch.split(getattr(lstm, f'weight_hh_l{i}'), lstm.hidden_size)
                M_hi, M_hf, M_hg, M_ho = torch.split(mask, lstm.hidden_size)
                W_hi = W_hi[M_hi][:, M_hi]
                W_hf = W_hf[M_hf][:, M_hf]
                W_hg = W_hg[M_hg][:, M_hg]
                W_ho = W_ho[M_ho][:, M_ho]
                new_weight = torch.cat((W_hi, W_hf, W_hg, W_ho))
                setattr(lstm, f'weight_hh_l{i}', nn.Parameter(new_weight))
                setattr(lstm, f'bias_hh_l{i}', nn.Parameter(getattr(lstm, f'bias_hh_l{i}')[mask]))
            if i + 1 == lstm.num_layers:
                lstm.hidden_size = int(M_hi.sum())
                with torch.no_grad():
                    if parametrize.is_parametrized(linear):
                        parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
                        weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
                        weight_parameterizations.original = nn.Parameter(weight_parameterizations.original[:, M_ho])
                        linear.in_features = weight_parameterizations.original.shape[1]
                    else:
                        linear.weight = nn.Parameter(linear.weight[:, M_ho])
                        linear.in_features = linear.weight.shape[1]
                    if layernorm is not None:
                        layernorm.normalized_shape = (linear.in_features,)
                        layernorm.weight = nn.Parameter(layernorm.weight[M_ho])
                        layernorm.bias = nn.Parameter(layernorm.bias[M_ho])
            else:
                with torch.no_grad():
                    if parametrize.is_parametrized(lstm, f'weight_ih_l{i + 1}'):
                        parametrization_dict = cast(nn.ModuleDict, lstm.parametrizations)
                        weight_parameterizations = cast(ParametrizationList, getattr(parametrization_dict, f'weight_ih_l{i + 1}'))
                        weight_parameterizations.original = nn.Parameter(weight_parameterizations.original[:, M_ho])
                    else:
                        next_layer_weight = getattr(lstm, f'weight_ih_l{i + 1}')
                        setattr(lstm, f'weight_ih_l{i + 1}', nn.Parameter(next_layer_weight[:, M_ho]))