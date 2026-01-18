from typing import Union
def mix_labellers(labellers, class_name='MixtureLabeller'):
    """Combine Labeller classes dynamically.

    The Labeller class aims to split plot labeling in ArviZ into atomic tasks to maximize
    extensibility, and the few classes provided are designed with small deviations
    from the base class, in many cases only one method is modified by the child class.
    It is to be expected then to want to use multiple classes "at once".

    This functions helps combine classes dynamically.

    For a general overview of ArviZ label customization, including
    ``mix_labellers``, see the :ref:`label_guide` page.

    Parameters
    ----------
    labellers : iterable of types
        Iterable of Labeller types to combine
    class_name : str, optional
        The name of the generated class

    Returns
    -------
        type
            Mixture class object. **It is not initialized**, and it should be
            initialized before passing it to ArviZ functions.

    Examples
    --------
    Combine the :class:`~arviz.labels.DimCoordLabeller` with the
    :class:`~arviz.labels.MapLabeller` to generate labels in the style of the
    ``DimCoordLabeller`` but using the mappings defined by ``MapLabeller``.
    Note that this works even though both modify the same methods because
    ``MapLabeller`` implements the mapping and then calls `super().method`.

    .. jupyter-execute::

        from arviz.labels import mix_labellers, DimCoordLabeller, MapLabeller
        l1 = DimCoordLabeller()
        sel = {"dim1": "a", "dim2": "top"}
        print(f"Output of DimCoordLabeller alone > {l1.sel_to_str(sel, sel)}")
        l2 = MapLabeller(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
        print(f"Output of MapLabeller alone > {l2.sel_to_str(sel, sel)}")
        l3 = mix_labellers(
            (MapLabeller, DimCoordLabeller)
        )(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
        print(f"Output of mixture labeller > {l3.sel_to_str(sel, sel)}")

    We can see how the mappings are taken into account as well as the dim+coord style. However,
    he order in the ``labellers`` arg iterator is important! See for yourself:

    .. jupyter-execute::

        l4 = mix_labellers(
            (DimCoordLabeller, MapLabeller)
        )(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
        print(f"Output of inverted mixture labeller > {l4.sel_to_str(sel, sel)}")

    """
    return type(class_name, labellers, {})