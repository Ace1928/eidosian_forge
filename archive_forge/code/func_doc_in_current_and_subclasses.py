from typing import Optional, TypeVar
def doc_in_current_and_subclasses(obj: T) -> T:
    """Overrides `do_not_doc_in_subclasses` decorator.

  If this decorator is set on a child class's method whose parent's method
  contains `do_not_doc_in_subclasses`, then that will be overriden and the
  child method will get documented. All classes inherting from the child will
  also document that method.

  For example:

  ```
  class Parent:
    @do_not_doc_in_subclasses
    def method1(self):
      pass
    def method2(self):
      pass

  class Child1(Parent):
    @doc_in_current_and_subclasses
    def method1(self):
      pass
    def method2(self):
      pass

  class Child2(Parent):
    def method1(self):
      pass
    def method2(self):
      pass

  class Child11(Child1):
    pass
  ```

  This will produce the following docs:

  ```
  /Parent.md
    # method1
    # method2
  /Child1.md
    # method1
    # method2
  /Child2.md
    # method2
  /Child11.md
    # method1
    # method2
  ```

  Args:
    obj: The class-attribute to hide from the generated docs.

  Returns:
    obj
  """
    setattr(obj, _DOC_IN_CURRENT_AND_SUBCLASSES, None)
    return obj