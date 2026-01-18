from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import logging
from fontTools.designspaceLib import (
def getStatNames(doc: DesignSpaceDocument, userLocation: SimpleLocationDict) -> StatNames:
    """Compute the family, style, PostScript names of the given ``userLocation``
    using the document's STAT information.

    Also computes localizations.

    If not enough STAT data is available for a given name, either its dict of
    localized names will be empty (family and style names), or the name will be
    None (PostScript name).

    .. versionadded:: 5.0
    """
    familyNames: Dict[str, str] = {}
    defaultSource: Optional[SourceDescriptor] = doc.findDefault()
    if defaultSource is None:
        LOGGER.warning('Cannot determine default source to look up family name.')
    elif defaultSource.familyName is None:
        LOGGER.warning("Cannot look up family name, assign the 'familyname' attribute to the default source.")
    else:
        familyNames = {'en': defaultSource.familyName, **defaultSource.localisedFamilyName}
    styleNames: Dict[str, str] = {}
    label = doc.labelForUserLocation(userLocation)
    if label is not None:
        styleNames = {'en': label.name, **label.labelNames}
    else:
        labels = _getAxisLabelsForUserLocation(doc.axes, userLocation)
        if labels:
            languages = set((language for label in labels for language in label.labelNames))
            languages.add('en')
            for language in languages:
                styleName = ' '.join((label.labelNames.get(language, label.defaultName) for label in labels if not label.elidable))
                if not styleName and doc.elidedFallbackName is not None:
                    styleName = doc.elidedFallbackName
                styleNames[language] = styleName
    if 'en' not in familyNames or 'en' not in styleNames:
        return StatNames(familyNames=familyNames, styleNames=styleNames, postScriptFontName=None, styleMapFamilyNames={}, styleMapStyleName=None)
    postScriptFontName = f'{familyNames['en']}-{styleNames['en']}'.replace(' ', '')
    styleMapStyleName, regularUserLocation = _getRibbiStyle(doc, userLocation)
    styleNamesForStyleMap = styleNames
    if regularUserLocation != userLocation:
        regularStatNames = getStatNames(doc, regularUserLocation)
        styleNamesForStyleMap = regularStatNames.styleNames
    styleMapFamilyNames = {}
    for language in set(familyNames).union(styleNames.keys()):
        familyName = familyNames.get(language, familyNames['en'])
        styleName = styleNamesForStyleMap.get(language, styleNamesForStyleMap['en'])
        styleMapFamilyNames[language] = (familyName + ' ' + styleName).strip()
    return StatNames(familyNames=familyNames, styleNames=styleNames, postScriptFontName=postScriptFontName, styleMapFamilyNames=styleMapFamilyNames, styleMapStyleName=styleMapStyleName)