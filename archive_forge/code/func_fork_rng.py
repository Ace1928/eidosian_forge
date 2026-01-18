import contextlib
from typing import Generator
import warnings
from torch._C import default_generator
import torch
@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, _caller='fork_rng', _devices_kw='devices', device_type='cuda') -> Generator:
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        deivce_type (str): device type str, default is `cuda`. As for custom device,
            see details in [Note: support the custom device with privateuse1]
    """
    device_type = torch.device(device_type).type
    device_mod = getattr(torch, device_type, None)
    if device_mod is None:
        raise RuntimeError(f'torch has no module of `{device_type}`, you should register ' + 'a module by `torch._register_device_module`.')
    global _fork_rng_warned_already
    if not enabled:
        yield
        return
    if devices is None:
        num_devices = device_mod.device_count()
        if num_devices > 1 and (not _fork_rng_warned_already):
            message = f"{device_type.upper()} reports that you have {num_devices} available devices, and you have used {_caller} without explicitly specifying which devices are being used. For safety, we initialize *every* {device_type.upper()} device by default, which can be quite slow if you have a lot of {device_type.upper()}s. If you know that you are only making use of a few {device_type.upper()} devices, set the environment variable {device_type.upper()}_VISIBLE_DEVICES or the '{_devices_kw}' keyword argument of {_caller} with the set of devices you are actually using. For example, if you are using CPU only, set device.upper()_VISIBLE_DEVICES= or devices=[]; if you are using device 0 only, set {device_type.upper()}_VISIBLE_DEVICES=0 or devices=[0].  To initialize all devices and suppress this warning, set the '{_devices_kw}' keyword argument to `range(torch.{device_type}.device_count())`."
            warnings.warn(message)
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        devices = list(devices)
    cpu_rng_state = torch.get_rng_state()
    device_rng_states = []
    for device in devices:
        device_rng_states.append(device_mod.get_rng_state(device))
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        for device, device_rng_state in zip(devices, device_rng_states):
            device_mod.set_rng_state(device_rng_state, device)