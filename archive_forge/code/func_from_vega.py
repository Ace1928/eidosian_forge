from dataclasses import dataclass
from typing import List, Dict, Any, NewType, Optional
@staticmethod
def from_vega(name: str, signal: Optional[Dict[str, list]], store: Store):
    """
        Construct an IntervalSelection from the raw Vega signal and dataset values.

        Parameters
        ----------
        name: str
            The selection's name
        signal: dict or None
            The value of the Vega signal corresponding to the selection
        store: list
            The value of the Vega dataset corresponding to the selection.
            This dataset is named "{name}_store" in the Vega view.

        Returns
        -------
        PointSelection
        """
    if signal is None:
        signal = {}
    return IntervalSelection(name=name, value=signal, store=store)