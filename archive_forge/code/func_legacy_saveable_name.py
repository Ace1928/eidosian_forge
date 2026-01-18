def legacy_saveable_name(name):
    """Decorator to set the local name to use in the Checkpoint.

  Needed for migrating certain Trackables (see next paragraph) from the legacy
  `_gather_saveables_for_checkpoint` to the new `_serialize_to_tensors`
  function.

  This decorator should be used if the SaveableObject generates tensors with
  different names from the name that is passed to the factory.

  Example migration:

  *Before*

  ```
  class MyTrackable(Trackable):
    def _gather_saveables_for_checkpoint(self):
      return {"key": _MySaveable}

  class _MySaveable(SaveableObject):
    def __init__(self, name):
      specs = [
          SaveSpec(tensor1, "", name + "-1")
          SaveSpec(tensor2, "", name + "-2")
      ]
      super().__init__(None, specs, name)
  ```

  *After*

  ```
  @legacy_saveable_name("key")
  class MyTrackable(Trackable):

    def _serialize_to_tensors(self):
      return {"key-1": tensor1, "key-2": tensor2}
  ```

  Args:
    name: String name of the SaveableObject factory (the key returned in the
       `_gather_saveables_for_checkpoint` function)

  Returns:
    A decorator.
  """

    def decorator(cls_or_obj):
        setattr(cls_or_obj, _LEGACY_SAVEABLE_NAME, name)
        return cls_or_obj
    return decorator