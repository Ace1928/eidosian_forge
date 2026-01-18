import ctypes
import enum
import os
import platform
import sys
import numpy as np
def get_signature_runner(self, signature_key=None):
    """Gets callable for inference of specific SignatureDef.

    Example usage,
    ```
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    fn = interpreter.get_signature_runner('div_with_remainder')
    output = fn(x=np.array([3]), y=np.array([2]))
    print(output)
    # {
    #   'quotient': array([1.], dtype=float32)
    #   'remainder': array([1.], dtype=float32)
    # }
    ```

    None can be passed for signature_key if the model has a single Signature
    only.

    All names used are this specific SignatureDef names.


    Args:
      signature_key: Signature key for the SignatureDef, it can be None if and
        only if the model has a single SignatureDef. Default value is None.

    Returns:
      This returns a callable that can run inference for SignatureDef defined
      by argument 'signature_key'.
      The callable will take key arguments corresponding to the arguments of the
      SignatureDef, that should have numpy values.
      The callable will returns dictionary that maps from output names to numpy
      values of the computed results.

    Raises:
      ValueError: If passed signature_key is invalid.
    """
    if signature_key is None:
        if len(self._signature_defs) != 1:
            raise ValueError('SignatureDef signature_key is None and model has {0} Signatures. None is only allowed when the model has 1 SignatureDef'.format(len(self._signature_defs)))
        else:
            signature_key = next(iter(self._signature_defs))
    return SignatureRunner(interpreter=self, signature_key=signature_key)