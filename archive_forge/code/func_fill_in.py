import tree
from keras.src.backend import KerasTensor
def fill_in(self, tensor_dict):
    """Maps KerasTensors to computed values using `tensor_dict`.

        `tensor_dict` maps `KerasTensor` instances to their current values.
        """
    if self._single_positional_tensor is not None:
        return ((tensor_dict[id(self._single_positional_tensor)],), {})

    def switch_fn(x):
        if isinstance(x, KerasTensor):
            val = tensor_dict.get(id(x), None)
            if val is not None:
                return val
        return x
    return self.convert(switch_fn)