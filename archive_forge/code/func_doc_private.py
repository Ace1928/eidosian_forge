from typing import Optional, TypeVar
def doc_private(obj: T) -> T:
    """A decorator: Generates docs for private methods/functions.

  For example:

  ```
  class Try:

    @doc_controls.doc_private
    def _private(self):
      ...
  ```

  As a rule of thumb, private(beginning with `_`) methods/functions are
  not documented.

  This decorator allows to force document a private method/function.

  Args:
    obj: The class-attribute to hide from the generated docs.

  Returns:
    obj
  """
    setattr(obj, _DOC_PRIVATE, None)
    return obj