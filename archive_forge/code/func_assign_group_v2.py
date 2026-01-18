from tensorflow.python.ops import gen_collective_ops
def assign_group_v2(group_assignment, device_index, base_key):
    """Assign group key based on group_assignment.

  Args:
    group_assignment: a 2 dimensional integer Tensor that encodes which devices
      belong to the same group. The values are indices of the devices within 0
      to number of devices.
    device_index: integer for the index of the current device
    base_key: integer to offset the resulted group_key. The base key shall be
      unique for different values of group_assignment in the same tf.function.
  Notes: The device_index argument must be consistent with the index of the
    device of this Op in the device assignment list. The behavior of this Op is
    undefined if they are inconsistent.

  Returns:
    group_size, group_key: The group size and group key for the current device.
  """
    group_size, group_key = gen_collective_ops.collective_assign_group_v2(group_assignment=group_assignment, device_index=device_index, base_key=base_key)
    return (group_size, group_key)